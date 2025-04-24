package ptithcm.entity;
import java.util.List;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.FetchType;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;
import javax.persistence.JoinColumn;
import javax.persistence.ManyToOne;
import javax.persistence.OneToMany;
import javax.persistence.Table;

@Entity
@Table(name="KIEUSANPHAM")
public class KieuSanPhamEntity{
	@Id
	@Column(name="MAKIEU")
	@GeneratedValue
	private int maKieu;
	@Column(name = "TENKIEU")
	private String tenKieu;
	@ManyToOne
	@JoinColumn(name = "MALOAI")
	private LoaiSanPhamEntity loai;
	
	@OneToMany(mappedBy = "maKieu", fetch = FetchType.EAGER)
	private List<SanPhamEntity> sanPhams;
	
	
	public int getMaKieu() {
		return maKieu;
	}
	public void setMaKieu(int maKieu) {
		this.maKieu = maKieu;
	}
	public String getTenKieu() {
		return tenKieu;
	}
	public void setTenKieu(String tenKieu) {
		this.tenKieu = tenKieu;
	}
	public LoaiSanPhamEntity getLoai() {
		return loai;
	}
	public void setLoai(LoaiSanPhamEntity loai) {
		this.loai = loai;
	}
	
}