package ptithcm.dao;

import java.util.List;

import javax.transaction.Transactional;

import org.hibernate.Query;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.Transaction;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import ptithcm.entity.KieuSanPhamEntity;

@Transactional
@Repository
public class KieuSanPhamDAOImpl implements KieuSanPhamDAO{
	
	@Autowired
	private SessionFactory sessionFactory;

	@Override
	public List<KieuSanPhamEntity> layKieu() {
		Session session = sessionFactory.getCurrentSession();
		String hql = "FROM KieuSanPhamEntity ORDER BY tenKieu ASC";
		Query query = session.createQuery(hql);
	    List<KieuSanPhamEntity> listKieu = query.list();
	    return listKieu;
	}
	
	@Override
	public List<KieuSanPhamEntity> layKieuTheoLoai(String loai){
		Session session = sessionFactory.getCurrentSession();
		String hql = "FROM KieuSanPhamEntity kieu WHERE kieu.maLoai =: loai";
		Query query = session.createQuery(hql).setParameter("loai", loai);
	    List<KieuSanPhamEntity> dsKieu = query.list();
	    return dsKieu;
	}
	@Override
	public KieuSanPhamEntity layKieuTheoMa(String maKieu) {
		Session session = sessionFactory.getCurrentSession();
	    KieuSanPhamEntity Kieu = (KieuSanPhamEntity) session.get(KieuSanPhamEntity.class, maKieu);
	    return Kieu;
	}

	@Override
	public KieuSanPhamEntity layKieuTheoMa(int maKieu) {
		Session session = sessionFactory.getCurrentSession();
	    KieuSanPhamEntity Kieu = (KieuSanPhamEntity) session.get(KieuSanPhamEntity.class, maKieu);
	    return Kieu;
	}

	@Override
	public void themKieu(KieuSanPhamEntity kieu) {
		Session session=sessionFactory.openSession();
		Transaction t = session.beginTransaction();
		try {
			session.save(kieu);
			t.commit();

		} catch (Exception ex) {
			t.rollback();
			System.out.print("loi");

		} finally {
			session.close();
		}
	}

	@Override
	public void updateKieu(KieuSanPhamEntity kieu) {
		Session session=sessionFactory.openSession();
		Transaction t = session.beginTransaction();
		try {
			session.update(kieu);
			t.commit();

		} catch (Exception ex) {
			t.rollback();
			System.out.print("loi");

		} finally {
			session.close();
		}		
	}

	@Override
	public void xoaKieu(KieuSanPhamEntity kieu) {
		sessionFactory.getCurrentSession().delete(kieu);
	}

}
