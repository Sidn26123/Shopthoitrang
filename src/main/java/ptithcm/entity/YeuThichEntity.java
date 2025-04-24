package ptithcm.entity;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;
import javax.persistence.JoinColumn;
import javax.persistence.ManyToOne;
import javax.persistence.Table;

@Entity
@Table(name="YEUTHICH")
public class YeuThichEntity {
	@Id
	@GeneratedValue
	@Column(name="ID")
	private int id;
	@ManyToOne
	@JoinColumn(name="MAND")
	private NguoiDungEntity nguoiDung;
	@ManyToOne
	@JoinColumn(name="MASP")
	private SanPhamEntity sanPham;
	public int getId() {
		return id;
	}
	public void setId(int id) {
		this.id = id;
	}
	public NguoiDungEntity getNguoiDung() {
		return nguoiDung;
	}
	public void setNguoiDung(NguoiDungEntity nguoiDung) {
		this.nguoiDung = nguoiDung;
	}
	public SanPhamEntity getSanPham() {
		return sanPham;
	}
	public void setSanPham(SanPhamEntity sanPham) {
		this.sanPham = sanPham;
	}
	
}
